// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use j2k_jpeg::{ColorSpace, Decoder, Downscale, PixelFormat, Rect};
use libfuzzer_sys::fuzz_target;

const MAX_OUTPUT_BYTES: usize = 1 << 20;

fuzz_target!(|data: &[u8]| {
    let Ok(decoder) = Decoder::new(data) else {
        return;
    };

    let info = decoder.info();
    let dims = info.dimensions;
    if dims.0 == 0 || dims.1 == 0 {
        return;
    }

    let scale = downscale_from(data.first().copied().unwrap_or_default());
    let fmt = pixel_format(info.color_space, info.bit_depth);
    let full_frame = data.get(1).copied().unwrap_or_default() & 1 == 0;
    let roi = bounded_rect(dims, data.get(2..).unwrap_or_default());
    let out_dims = if full_frame {
        scaled_dims(dims, scale)
    } else {
        scaled_rect_dims(roi, scale)
    };

    let Some((stride, len)) = output_geometry(out_dims, fmt) else {
        return;
    };
    if len == 0 || len > MAX_OUTPUT_BYTES {
        return;
    }

    let mut out = vec![0_u8; len];
    if full_frame {
        let _ = decoder.decode_scaled_into(&mut out, stride, fmt, scale);
    } else {
        let _ = decoder.decode_region_scaled_into(&mut out, stride, fmt, roi, scale);
    }
});

fn pixel_format(color_space: ColorSpace, bit_depth: u8) -> PixelFormat {
    match (color_space, bit_depth > 8) {
        (ColorSpace::Grayscale, false) => PixelFormat::Gray8,
        (ColorSpace::Grayscale, true) => PixelFormat::Gray16,
        (_, false) => PixelFormat::Rgb8,
        (_, true) => PixelFormat::Rgb16,
    }
}

fn downscale_from(value: u8) -> Downscale {
    match value & 3 {
        0 => Downscale::None,
        1 => Downscale::Half,
        2 => Downscale::Quarter,
        _ => Downscale::Eighth,
    }
}

fn bounded_rect(dims: (u32, u32), data: &[u8]) -> Rect {
    let x = bounded_u32(data.first().copied().unwrap_or_default(), dims.0);
    let y = bounded_u32(data.get(1).copied().unwrap_or_default(), dims.1);
    let max_w = dims.0.saturating_sub(x).max(1);
    let max_h = dims.1.saturating_sub(y).max(1);
    let w = 1 + bounded_u32(data.get(2).copied().unwrap_or_default(), max_w);
    let h = 1 + bounded_u32(data.get(3).copied().unwrap_or_default(), max_h);
    Rect { x, y, w, h }
}

fn bounded_u32(value: u8, max_exclusive: u32) -> u32 {
    if max_exclusive == 0 {
        0
    } else {
        u32::from(value) % max_exclusive
    }
}

fn scaled_dims(dims: (u32, u32), scale: Downscale) -> (u32, u32) {
    let denom = scale.denominator();
    (dims.0.div_ceil(denom), dims.1.div_ceil(denom))
}

fn scaled_rect_dims(rect: Rect, scale: Downscale) -> (u32, u32) {
    let denom = scale.denominator();
    let x1 = rect.x.saturating_add(rect.w).div_ceil(denom);
    let y1 = rect.y.saturating_add(rect.h).div_ceil(denom);
    let x0 = rect.x / denom;
    let y0 = rect.y / denom;
    (x1.saturating_sub(x0), y1.saturating_sub(y0))
}

fn output_geometry(dims: (u32, u32), fmt: PixelFormat) -> Option<(usize, usize)> {
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let len = stride.checked_mul(dims.1 as usize)?;
    Some((stride, len))
}
